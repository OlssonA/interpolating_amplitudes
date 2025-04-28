module     p0_ubaru_httbar_d1h6l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity6d1h6l1.f90
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
      use p0_ubaru_httbar_abbrevd1h6
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc1(17)
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspval5k1
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspk2
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspval5k1 = dotproduct(Q,spval5k1)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspk2 = dotproduct(Q,k2)
      acc1(1)=abb1(9)
      acc1(2)=abb1(10)
      acc1(3)=abb1(13)
      acc1(4)=abb1(16)
      acc1(5)=abb1(17)
      acc1(6)=abb1(19)
      acc1(7)=abb1(20)
      acc1(8)=abb1(22)
      acc1(9)=abb1(23)
      acc1(10)=abb1(30)
      acc1(11)=abb1(31)
      acc1(12)=acc1(3)*Qspvak2l4
      acc1(13)=acc1(5)*Qspvak2l3
      acc1(12)=acc1(8)+acc1(13)+acc1(12)
      acc1(12)=Qspval5k1*acc1(12)
      acc1(13)=acc1(2)*Qspvak2k1
      acc1(14)=acc1(10)*Qspval3k1
      acc1(13)=acc1(14)+acc1(9)+acc1(13)
      acc1(13)=Qspk2*acc1(13)
      acc1(14)=acc1(1)*Qspvak2k1
      acc1(15)=acc1(4)*Qspval3k1
      acc1(16)=acc1(6)*Qspvak2l4
      acc1(17)=acc1(11)*Qspvak2l3
      brack=acc1(7)+acc1(12)+acc1(13)+acc1(14)+acc1(15)+acc1(16)+acc1(17)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_ubaru_httbar_d1h6l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd1h6
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d1
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k5
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d1 = 0.0_ki
      d1 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d1, ki), aimag(d1), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_ubaru_httbar_d1h6l1
