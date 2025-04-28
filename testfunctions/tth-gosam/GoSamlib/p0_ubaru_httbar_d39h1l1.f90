module     p0_ubaru_httbar_d39h1l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity1d39h1l1.f90
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
      use p0_ubaru_httbar_abbrevd39h1
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc39(18)
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspk1
      complex(ki) :: Qspk2
      complex(ki) :: Qspvak1l3
      complex(ki) :: QspQ
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspk1 = dotproduct(Q,k1)
      Qspk2 = dotproduct(Q,k2)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      QspQ = dotproduct(Q,Q)
      acc39(1)=abb39(10)
      acc39(2)=abb39(11)
      acc39(3)=abb39(12)
      acc39(4)=abb39(13)
      acc39(5)=abb39(15)
      acc39(6)=abb39(16)
      acc39(7)=abb39(17)
      acc39(8)=abb39(18)
      acc39(9)=abb39(19)
      acc39(10)=abb39(21)
      acc39(11)=abb39(22)
      acc39(12)=abb39(26)
      acc39(13)=acc39(7)*Qspval4k2
      acc39(14)=acc39(9)*Qspval5k2
      acc39(15)=Qspval5l3*acc39(6)
      acc39(16)=Qspval4l3*acc39(12)
      acc39(17)=Qspval3k2*acc39(4)
      acc39(13)=acc39(17)+acc39(16)+acc39(15)+acc39(14)+acc39(2)+acc39(13)
      acc39(13)=Qspvak1k2*acc39(13)
      acc39(14)=-Qspk1-Qspk2
      acc39(14)=acc39(5)*acc39(14)
      acc39(15)=acc39(10)*Qspval4k2
      acc39(16)=acc39(11)*Qspval5k2
      acc39(17)=Qspvak1l3*acc39(1)
      acc39(18)=QspQ*acc39(3)
      brack=acc39(8)+acc39(13)+acc39(14)+acc39(15)+acc39(16)+acc39(17)+acc39(18)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_ubaru_httbar_d39h1l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd39h1
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d39
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k3-k4-k5
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d39 = 0.0_ki
      d39 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d39, ki), aimag(d39), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_ubaru_httbar_d39h1l1
