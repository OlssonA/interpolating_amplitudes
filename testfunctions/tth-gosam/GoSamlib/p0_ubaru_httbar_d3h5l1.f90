module     p0_ubaru_httbar_d3h5l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity5d3h5l1.f90
   ! generator: buildfortran.py
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_util, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd3h5
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc3(17)
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspk2
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspk2 = dotproduct(Q,k2)
      acc3(1)=abb3(9)
      acc3(2)=abb3(10)
      acc3(3)=abb3(13)
      acc3(4)=abb3(16)
      acc3(5)=abb3(17)
      acc3(6)=abb3(19)
      acc3(7)=abb3(20)
      acc3(8)=abb3(22)
      acc3(9)=abb3(23)
      acc3(10)=abb3(26)
      acc3(11)=abb3(30)
      acc3(12)=acc3(3)*Qspval5k2
      acc3(13)=acc3(5)*Qspval3k2
      acc3(12)=acc3(8)+acc3(13)+acc3(12)
      acc3(12)=Qspvak1l4*acc3(12)
      acc3(13)=acc3(2)*Qspvak1k2
      acc3(14)=acc3(11)*Qspvak1l3
      acc3(13)=acc3(14)+acc3(9)+acc3(13)
      acc3(13)=Qspk2*acc3(13)
      acc3(14)=acc3(1)*Qspvak1k2
      acc3(15)=acc3(4)*Qspvak1l3
      acc3(16)=acc3(6)*Qspval5k2
      acc3(17)=acc3(10)*Qspval3k2
      brack=acc3(7)+acc3(12)+acc3(13)+acc3(14)+acc3(15)+acc3(16)+acc3(17)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_ubaru_httbar_d3h5l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd3h5
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d3
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k4
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d3 = 0.0_ki
      d3 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d3, ki), aimag(d3), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_ubaru_httbar_d3h5l1
