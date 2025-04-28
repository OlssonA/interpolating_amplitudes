module     p2_gg_httbar_d86h0l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d86h0l1_qp.f90
   ! generator: buildfortran.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd86h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc86(69)
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspval3l5
      complex(ki) :: QspQ
      complex(ki) :: Qspl5
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspk2
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspe1
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspval3l5 = dotproduct(Q,spval3l5)
      QspQ = dotproduct(Q,Q)
      Qspl5 = dotproduct(Q,l5)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspk2 = dotproduct(Q,k2)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspe1 = dotproduct(Q,e1)
      acc86(1)=abb86(8)
      acc86(2)=abb86(9)
      acc86(3)=abb86(10)
      acc86(4)=abb86(11)
      acc86(5)=abb86(12)
      acc86(6)=abb86(13)
      acc86(7)=abb86(14)
      acc86(8)=abb86(15)
      acc86(9)=abb86(16)
      acc86(10)=abb86(17)
      acc86(11)=abb86(18)
      acc86(12)=abb86(19)
      acc86(13)=abb86(20)
      acc86(14)=abb86(21)
      acc86(15)=abb86(22)
      acc86(16)=abb86(23)
      acc86(17)=abb86(24)
      acc86(18)=abb86(25)
      acc86(19)=abb86(26)
      acc86(20)=abb86(27)
      acc86(21)=abb86(28)
      acc86(22)=abb86(29)
      acc86(23)=abb86(30)
      acc86(24)=abb86(31)
      acc86(25)=abb86(32)
      acc86(26)=abb86(33)
      acc86(27)=abb86(34)
      acc86(28)=abb86(35)
      acc86(29)=abb86(36)
      acc86(30)=abb86(37)
      acc86(31)=abb86(38)
      acc86(32)=abb86(39)
      acc86(33)=abb86(40)
      acc86(34)=abb86(41)
      acc86(35)=abb86(42)
      acc86(36)=abb86(43)
      acc86(37)=abb86(44)
      acc86(38)=abb86(45)
      acc86(39)=abb86(46)
      acc86(40)=abb86(48)
      acc86(41)=abb86(51)
      acc86(42)=abb86(53)
      acc86(43)=abb86(55)
      acc86(44)=abb86(58)
      acc86(45)=abb86(63)
      acc86(46)=abb86(68)
      acc86(47)=abb86(69)
      acc86(48)=abb86(70)
      acc86(49)=abb86(73)
      acc86(50)=acc86(16)*Qspval5k2
      acc86(51)=acc86(17)*Qspvae2k2
      acc86(52)=acc86(22)*Qspval4l5
      acc86(53)=acc86(25)*Qspval3k2
      acc86(54)=acc86(26)*Qspval4k2
      acc86(55)=acc86(27)*Qspvae2k1
      acc86(56)=acc86(31)*Qspval3l5
      acc86(57)=acc86(38)*QspQ
      acc86(58)=-acc86(40)*Qspl5
      acc86(59)=acc86(45)*Qspvae2l3
      acc86(50)=acc86(59)+acc86(58)+acc86(57)+acc86(56)+acc86(55)+acc86(54)+acc&
      &86(53)+acc86(52)+acc86(21)+acc86(51)+acc86(50)
      acc86(50)=Qspvae1e2*acc86(50)
      acc86(51)=acc86(7)*Qspval5k2
      acc86(52)=acc86(8)*Qspvak1e2
      acc86(53)=acc86(18)*Qspk2
      acc86(54)=acc86(29)*Qspval5e2
      acc86(55)=acc86(32)*Qspval5l3
      acc86(56)=acc86(37)*QspQ
      acc86(57)=acc86(39)*Qspvak2l3
      acc86(58)=acc86(43)*Qspval4e2
      acc86(59)=acc86(49)*Qspval3e2
      acc86(51)=acc86(59)+acc86(58)+acc86(57)+acc86(56)+acc86(36)+acc86(55)+acc&
      &86(54)+acc86(53)+acc86(52)+acc86(51)
      acc86(51)=Qspvae2e1*acc86(51)
      acc86(52)=acc86(1)*Qspval5e2
      acc86(53)=acc86(5)*Qspval4e2
      acc86(54)=acc86(24)*Qspval3e2
      acc86(52)=acc86(54)+acc86(53)+acc86(4)+acc86(52)
      acc86(52)=acc86(52)*Qspvae2k2
      acc86(53)=acc86(46)*Qspvae2l3
      acc86(53)=acc86(53)+acc86(10)
      acc86(53)=acc86(53)*Qspval5e2
      acc86(54)=acc86(2)*Qspval4e2
      acc86(55)=acc86(3)*Qspvae2l3
      acc86(56)=acc86(48)*Qspval3e2
      acc86(52)=acc86(56)+acc86(14)+acc86(55)+acc86(54)+acc86(52)+acc86(53)
      acc86(52)=Qspe1*acc86(52)
      acc86(53)=acc86(9)*Qspvak1e2
      acc86(54)=acc86(11)*Qspvae2k2
      acc86(55)=acc86(12)*Qspval5k2
      acc86(56)=acc86(13)*Qspval3k2
      acc86(57)=acc86(15)*QspQ
      acc86(58)=acc86(19)*Qspval4l5
      acc86(59)=acc86(20)*Qspval5e2
      acc86(60)=acc86(23)*Qspval5l3
      acc86(61)=acc86(28)*Qspval3l5
      acc86(62)=acc86(30)*Qspval4k2
      acc86(63)=acc86(33)*Qspl5
      acc86(64)=acc86(34)*Qspvak2l3
      acc86(65)=acc86(35)*Qspvae2k1
      acc86(66)=acc86(41)*Qspk2
      acc86(67)=acc86(42)*Qspval4e2
      acc86(68)=acc86(44)*Qspvae2l3
      acc86(69)=acc86(47)*Qspval3e2
      brack=acc86(6)+acc86(50)+acc86(51)+acc86(52)+acc86(53)+acc86(54)+acc86(55&
      &)+acc86(56)+acc86(57)+acc86(58)+acc86(59)+acc86(60)+acc86(61)+acc86(62)+&
      &acc86(63)+acc86(64)+acc86(65)+acc86(66)+acc86(67)+acc86(68)+acc86(69)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d86h0l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd86h0_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d86
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k2+k3+k4+k5
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d86 = 0.0_ki
      d86 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d86, ki), aimag(d86), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d86h0l1_qp
