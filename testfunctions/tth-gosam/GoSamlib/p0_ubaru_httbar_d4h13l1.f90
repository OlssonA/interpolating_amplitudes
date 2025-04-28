module     p0_ubaru_httbar_d4h13l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity13d4h13l1.f90
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
      use p0_ubaru_httbar_abbrevd4h13
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc4(31)
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspl4
      complex(ki) :: Qspk2
      complex(ki) :: Qspk1
      complex(ki) :: QspQ
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspl4 = dotproduct(Q,l4)
      Qspk2 = dotproduct(Q,k2)
      Qspk1 = dotproduct(Q,k1)
      QspQ = dotproduct(Q,Q)
      acc4(1)=abb4(9)
      acc4(2)=abb4(10)
      acc4(3)=abb4(11)
      acc4(4)=abb4(12)
      acc4(5)=abb4(13)
      acc4(6)=abb4(14)
      acc4(7)=abb4(15)
      acc4(8)=abb4(16)
      acc4(9)=abb4(17)
      acc4(10)=abb4(18)
      acc4(11)=abb4(19)
      acc4(12)=abb4(20)
      acc4(13)=abb4(21)
      acc4(14)=abb4(22)
      acc4(15)=abb4(26)
      acc4(16)=abb4(31)
      acc4(17)=abb4(34)
      acc4(18)=abb4(35)
      acc4(19)=acc4(4)*Qspvak2l4
      acc4(20)=acc4(7)*Qspval3l4
      acc4(21)=Qspvak2l5*acc4(11)
      acc4(22)=Qspvak2l3*acc4(14)
      acc4(19)=acc4(22)+acc4(21)+acc4(20)+acc4(5)+acc4(19)
      acc4(19)=Qspvak1k2*acc4(19)
      acc4(20)=acc4(3)*Qspval3l4
      acc4(21)=acc4(9)*Qspvak2l4
      acc4(22)=Qspval4l5*acc4(17)
      acc4(23)=Qspval4l3*acc4(13)
      acc4(24)=Qspval3k2*acc4(15)
      acc4(25)=Qspvak1l5*acc4(16)
      acc4(26)=Qspvak1l4*acc4(18)
      acc4(27)=Qspvak1l3*acc4(12)
      acc4(28)=Qspl4*acc4(6)
      acc4(29)=Qspk2*acc4(2)
      acc4(30)=Qspk1*acc4(8)
      acc4(31)=QspQ*acc4(10)
      brack=acc4(1)+acc4(19)+acc4(20)+acc4(21)+acc4(22)+acc4(23)+acc4(24)+acc4(&
      &25)+acc4(26)+acc4(27)+acc4(28)+acc4(29)+acc4(30)+acc4(31)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_ubaru_httbar_d4h13l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd4h13
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d4
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k3+k4+k5
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d4 = 0.0_ki
      d4 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d4, ki), aimag(d4), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_ubaru_httbar_d4h13l1
