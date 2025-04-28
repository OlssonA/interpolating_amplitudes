module     p0_ubaru_httbar_d66h5l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity5d66h5l1.f90
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
      use p0_ubaru_httbar_abbrevd66h5
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc66(18)
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspk2
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspval5l4
      complex(ki) :: QspQ
      complex(ki) :: Qspk1
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspk2 = dotproduct(Q,k2)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspval5l4 = dotproduct(Q,spval5l4)
      QspQ = dotproduct(Q,Q)
      Qspk1 = dotproduct(Q,k1)
      acc66(1)=abb66(7)
      acc66(2)=abb66(8)
      acc66(3)=abb66(9)
      acc66(4)=abb66(11)
      acc66(5)=abb66(12)
      acc66(6)=abb66(14)
      acc66(7)=abb66(15)
      acc66(8)=abb66(23)
      acc66(9)=abb66(25)
      acc66(10)=abb66(27)
      acc66(11)=acc66(6)*Qspvak2l4
      acc66(12)=acc66(5)*Qspval5k2
      acc66(13)=acc66(4)*Qspk2
      acc66(14)=acc66(2)*Qspval3k2
      acc66(15)=acc66(1)*Qspvak1l4
      acc66(16)=-Qspvak1k2*acc66(9)
      acc66(16)=acc66(10)+acc66(16)
      acc66(16)=Qspval3l4*acc66(16)
      acc66(17)=Qspvak1k2*acc66(3)
      acc66(17)=acc66(8)+acc66(17)
      acc66(17)=Qspval5l4*acc66(17)
      acc66(18)=-QspQ+Qspk1
      acc66(18)=acc66(7)*acc66(18)
      brack=acc66(11)+acc66(12)+acc66(13)+acc66(14)+acc66(15)+acc66(16)+acc66(1&
      &7)+acc66(18)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_ubaru_httbar_d66h5l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd66h5
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d66
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d66 = 0.0_ki
      d66 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d66, ki), aimag(d66), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_ubaru_httbar_d66h5l1
