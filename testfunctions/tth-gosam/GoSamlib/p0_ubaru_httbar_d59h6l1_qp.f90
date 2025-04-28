module     p0_ubaru_httbar_d59h6l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity6d59h6l1_qp.f90
   ! generator: buildfortran.py
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_util_qp, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd59h6_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc59(32)
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspk2
      complex(ki) :: QspQ
      complex(ki) :: Qspval5k1
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspl5
      complex(ki) :: Qspl4
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspk2 = dotproduct(Q,k2)
      QspQ = dotproduct(Q,Q)
      Qspval5k1 = dotproduct(Q,spval5k1)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspl5 = dotproduct(Q,l5)
      Qspl4 = dotproduct(Q,l4)
      acc59(1)=abb59(10)
      acc59(2)=abb59(11)
      acc59(3)=abb59(12)
      acc59(4)=abb59(13)
      acc59(5)=abb59(14)
      acc59(6)=abb59(15)
      acc59(7)=abb59(16)
      acc59(8)=abb59(17)
      acc59(9)=abb59(18)
      acc59(10)=abb59(19)
      acc59(11)=abb59(20)
      acc59(12)=abb59(21)
      acc59(13)=abb59(22)
      acc59(14)=abb59(24)
      acc59(15)=abb59(27)
      acc59(16)=abb59(31)
      acc59(17)=abb59(34)
      acc59(18)=abb59(38)
      acc59(19)=abb59(46)
      acc59(20)=acc59(4)*Qspvak2l4
      acc59(21)=acc59(9)*Qspvak2k1
      acc59(22)=-acc59(14)*Qspk2
      acc59(20)=acc59(22)+acc59(21)+acc59(20)+acc59(1)
      acc59(20)=QspQ*acc59(20)
      acc59(21)=acc59(10)*Qspvak2k1
      acc59(21)=acc59(12)+acc59(21)
      acc59(21)=Qspk2*acc59(21)
      acc59(22)=acc59(5)*Qspvak2l4
      acc59(22)=acc59(15)+acc59(22)
      acc59(22)=Qspval5k1*acc59(22)
      acc59(23)=acc59(2)*Qspvak2l4
      acc59(24)=acc59(6)*Qspvak2k1
      acc59(25)=Qspval5l4*acc59(11)
      acc59(26)=Qspval5k2*acc59(7)
      acc59(27)=Qspval4k2*acc59(16)
      acc59(28)=Qspval3l4*acc59(18)
      acc59(29)=Qspvak2l5*acc59(13)
      acc59(30)=Qspvak2l3*acc59(8)
      acc59(31)=Qspl5*acc59(17)
      acc59(32)=Qspl4*acc59(19)
      brack=acc59(3)+acc59(20)+acc59(21)+acc59(22)+acc59(23)+acc59(24)+acc59(25&
      &)+acc59(26)+acc59(27)+acc59(28)+acc59(29)+acc59(30)+acc59(31)+acc59(32)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_ubaru_httbar_d59h6l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd59h6_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d59
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k5
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d59 = 0.0_ki
      d59 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d59, ki), aimag(d59), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_ubaru_httbar_d59h6l1_qp
