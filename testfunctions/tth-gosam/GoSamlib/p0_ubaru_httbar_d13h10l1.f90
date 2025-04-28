module     p0_ubaru_httbar_d13h10l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity10d13h10l1.f90
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
      use p0_ubaru_httbar_abbrevd13h10
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc13(17)
      complex(ki) :: Qspk2
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2l3
      complex(ki) :: QspQ
      Qspk2 = dotproduct(Q,k2)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      QspQ = dotproduct(Q,Q)
      acc13(1)=abb13(9)
      acc13(2)=abb13(10)
      acc13(3)=abb13(11)
      acc13(4)=abb13(12)
      acc13(5)=abb13(13)
      acc13(6)=abb13(15)
      acc13(7)=abb13(16)
      acc13(8)=abb13(17)
      acc13(9)=abb13(18)
      acc13(10)=abb13(22)
      acc13(11)=acc13(6)*Qspk2
      acc13(12)=Qspval4l5*acc13(3)
      acc13(11)=acc13(12)+acc13(11)+acc13(1)
      acc13(11)=Qspvak2k1*acc13(11)
      acc13(12)=acc13(4)*Qspk2
      acc13(13)=Qspval4k1*acc13(8)
      acc13(14)=Qspval3k1*acc13(10)
      acc13(15)=Qspvak2l5*acc13(7)
      acc13(16)=Qspvak2l3*acc13(9)
      acc13(17)=QspQ*acc13(5)
      brack=acc13(2)+acc13(11)+acc13(12)+acc13(13)+acc13(14)+acc13(15)+acc13(16&
      &)+acc13(17)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_ubaru_httbar_d13h10l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd13h10
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d13
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k3+k4
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d13 = 0.0_ki
      d13 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d13, ki), aimag(d13), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_ubaru_httbar_d13h10l1
