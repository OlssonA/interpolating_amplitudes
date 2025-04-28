module     p2_gg_httbar_d71h4l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d71h4l1_qp.f90
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
      use p2_gg_httbar_abbrevd71h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc71(113)
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspk2
      complex(ki) :: Qspl5
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspval5k1
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspval5k2
      complex(ki) :: QspQ
      complex(ki) :: Qspe2
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvak2l5
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspk2 = dotproduct(Q,k2)
      Qspl5 = dotproduct(Q,l5)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspval5k1 = dotproduct(Q,spval5k1)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspval5k2 = dotproduct(Q,spval5k2)
      QspQ = dotproduct(Q,Q)
      Qspe2 = dotproduct(Q,e2)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      acc71(1)=abb71(9)
      acc71(2)=abb71(10)
      acc71(3)=abb71(11)
      acc71(4)=abb71(12)
      acc71(5)=abb71(13)
      acc71(6)=abb71(14)
      acc71(7)=abb71(15)
      acc71(8)=abb71(16)
      acc71(9)=abb71(17)
      acc71(10)=abb71(18)
      acc71(11)=abb71(19)
      acc71(12)=abb71(20)
      acc71(13)=abb71(21)
      acc71(14)=abb71(22)
      acc71(15)=abb71(23)
      acc71(16)=abb71(24)
      acc71(17)=abb71(25)
      acc71(18)=abb71(26)
      acc71(19)=abb71(27)
      acc71(20)=abb71(28)
      acc71(21)=abb71(29)
      acc71(22)=abb71(30)
      acc71(23)=abb71(31)
      acc71(24)=abb71(32)
      acc71(25)=abb71(33)
      acc71(26)=abb71(34)
      acc71(27)=abb71(35)
      acc71(28)=abb71(36)
      acc71(29)=abb71(37)
      acc71(30)=abb71(38)
      acc71(31)=abb71(39)
      acc71(32)=abb71(40)
      acc71(33)=abb71(41)
      acc71(34)=abb71(42)
      acc71(35)=abb71(43)
      acc71(36)=abb71(44)
      acc71(37)=abb71(45)
      acc71(38)=abb71(46)
      acc71(39)=abb71(47)
      acc71(40)=abb71(48)
      acc71(41)=abb71(49)
      acc71(42)=abb71(50)
      acc71(43)=abb71(51)
      acc71(44)=abb71(52)
      acc71(45)=abb71(53)
      acc71(46)=abb71(54)
      acc71(47)=abb71(55)
      acc71(48)=abb71(56)
      acc71(49)=abb71(57)
      acc71(50)=abb71(58)
      acc71(51)=abb71(59)
      acc71(52)=abb71(60)
      acc71(53)=abb71(61)
      acc71(54)=abb71(62)
      acc71(55)=abb71(65)
      acc71(56)=abb71(67)
      acc71(57)=abb71(70)
      acc71(58)=abb71(75)
      acc71(59)=abb71(78)
      acc71(60)=abb71(80)
      acc71(61)=abb71(81)
      acc71(62)=abb71(83)
      acc71(63)=abb71(84)
      acc71(64)=abb71(85)
      acc71(65)=abb71(87)
      acc71(66)=abb71(89)
      acc71(67)=abb71(91)
      acc71(68)=abb71(93)
      acc71(69)=abb71(97)
      acc71(70)=abb71(102)
      acc71(71)=abb71(103)
      acc71(72)=abb71(105)
      acc71(73)=abb71(113)
      acc71(74)=abb71(122)
      acc71(75)=abb71(126)
      acc71(76)=Qspval5l3*acc71(20)
      acc71(77)=Qspk2*acc71(9)
      acc71(78)=Qspl5*acc71(71)
      acc71(79)=Qspvak1k2*acc71(51)
      acc71(80)=Qspvak1l3*acc71(48)
      acc71(81)=Qspval3k1*acc71(38)
      acc71(82)=Qspval3k2*acc71(29)
      acc71(83)=Qspval3l4*acc71(46)
      acc71(84)=Qspval4k2*acc71(36)
      acc71(85)=Qspval4l3*acc71(31)
      acc71(86)=Qspval5k1*acc71(16)
      acc71(87)=Qspval5l4*acc71(66)
      acc71(88)=Qspvae1k2*acc71(72)
      acc71(89)=Qspval3e1*acc71(75)
      acc71(90)=Qspvae1l3*acc71(56)
      acc71(91)=Qspval5e1*acc71(59)
      acc71(92)=Qspval5k2*acc71(10)
      acc71(93)=QspQ*acc71(19)
      acc71(76)=acc71(93)+acc71(92)+acc71(91)+acc71(90)+acc71(89)+acc71(88)+acc&
      &71(87)+acc71(86)+acc71(85)+acc71(84)+acc71(83)+acc71(82)+acc71(81)+acc71&
      &(80)+acc71(79)+acc71(78)+acc71(77)+acc71(18)+acc71(76)
      acc71(76)=Qspe2*acc71(76)
      acc71(77)=-acc71(69)*Qspvae2l4
      acc71(78)=-acc71(64)*Qspvae1e2
      acc71(79)=acc71(63)*Qspvak1e2
      acc71(80)=-acc71(62)*Qspval4e2
      acc71(81)=acc71(57)*Qspvae2k1
      acc71(82)=acc71(30)*Qspvae2e1
      acc71(83)=Qspvae2k2*acc71(61)
      acc71(84)=Qspval5e2*acc71(45)
      acc71(77)=acc71(84)+acc71(83)+acc71(82)+acc71(81)+acc71(80)+acc71(79)+acc&
      &71(78)+acc71(14)+acc71(77)
      acc71(77)=QspQ*acc71(77)
      acc71(78)=Qspvak1k2*acc71(53)
      acc71(79)=Qspval4k2*acc71(37)
      acc71(80)=Qspvae1k2*acc71(74)
      acc71(81)=-Qspval5k2*acc71(11)
      acc71(78)=acc71(81)+acc71(80)+acc71(79)+acc71(65)+acc71(78)
      acc71(78)=Qspvak2e2*acc71(78)
      acc71(79)=Qspval3k2*acc71(33)
      acc71(80)=-acc71(69)*Qspval3l4
      acc71(81)=acc71(57)*Qspval3k1
      acc71(82)=acc71(30)*Qspval3e1
      acc71(79)=acc71(82)+acc71(81)+acc71(80)+acc71(35)+acc71(79)
      acc71(79)=Qspvae2l3*acc71(79)
      acc71(80)=-acc71(69)*Qspval5l4
      acc71(81)=acc71(57)*Qspval5k1
      acc71(82)=acc71(30)*Qspval5e1
      acc71(83)=Qspval5k2*acc71(61)
      acc71(80)=acc71(83)+acc71(82)+acc71(81)+acc71(40)+acc71(80)
      acc71(80)=Qspvae2l5*acc71(80)
      acc71(81)=Qspvak2k1*acc71(47)
      acc71(82)=Qspvak2l4*acc71(44)
      acc71(83)=Qspvak2e1*acc71(54)
      acc71(84)=Qspk2*acc71(27)
      acc71(81)=acc71(84)+acc71(83)+acc71(82)+acc71(1)+acc71(81)
      acc71(81)=Qspvae2k2*acc71(81)
      acc71(82)=-Qspl5*acc71(23)
      acc71(83)=-acc71(64)*Qspvae1l5
      acc71(84)=acc71(63)*Qspvak1l5
      acc71(85)=-acc71(62)*Qspval4l5
      acc71(82)=acc71(85)+acc71(84)+acc71(83)+acc71(6)+acc71(82)
      acc71(82)=Qspval5e2*acc71(82)
      acc71(83)=-acc71(64)*Qspvae1l3
      acc71(84)=acc71(63)*Qspvak1l3
      acc71(85)=-acc71(62)*Qspval4l3
      acc71(83)=acc71(85)+acc71(84)+acc71(22)+acc71(83)
      acc71(83)=Qspval3e2*acc71(83)
      acc71(84)=acc71(39)*Qspvak2l5
      acc71(85)=Qspvak1l5*acc71(3)
      acc71(86)=Qspvak2k1*acc71(2)
      acc71(87)=Qspvak2l4*acc71(43)
      acc71(88)=Qspval4l5*acc71(12)
      acc71(89)=Qspval5l3*acc71(28)
      acc71(90)=Qspvak1e2*acc71(7)
      acc71(91)=Qspvae2k1*acc71(55)
      acc71(92)=Qspvak2e1*acc71(17)
      acc71(93)=Qspval4e2*acc71(70)
      acc71(94)=Qspvae2l4*acc71(68)
      acc71(95)=Qspvae1l5*acc71(49)
      acc71(96)=Qspvae1e2*acc71(25)
      acc71(97)=Qspvae2e1*acc71(26)
      acc71(98)=Qspk2*acc71(4)
      acc71(99)=Qspl5*acc71(41)
      acc71(100)=Qspvak1k2*acc71(50)
      acc71(101)=Qspvak1l3*acc71(21)
      acc71(102)=Qspval3k1*acc71(34)
      acc71(103)=Qspval3k2*acc71(13)
      acc71(104)=Qspval3l4*acc71(42)
      acc71(105)=Qspval4k2*acc71(32)
      acc71(106)=Qspval4l3*acc71(24)
      acc71(107)=Qspval5k1*acc71(15)
      acc71(108)=Qspval5l4*acc71(60)
      acc71(109)=Qspvae1k2*acc71(73)
      acc71(110)=Qspval3e1*acc71(52)
      acc71(111)=Qspvae1l3*acc71(67)
      acc71(112)=Qspval5e1*acc71(58)
      acc71(113)=Qspval5k2*acc71(5)
      brack=acc71(8)+acc71(76)+acc71(77)+acc71(78)+acc71(79)+acc71(80)+acc71(81&
      &)+acc71(82)+acc71(83)+acc71(84)+acc71(85)+acc71(86)+acc71(87)+acc71(88)+&
      &acc71(89)+acc71(90)+acc71(91)+acc71(92)+acc71(93)+acc71(94)+acc71(95)+ac&
      &c71(96)+acc71(97)+acc71(98)+acc71(99)+acc71(100)+acc71(101)+acc71(102)+a&
      &cc71(103)+acc71(104)+acc71(105)+acc71(106)+acc71(107)+acc71(108)+acc71(1&
      &09)+acc71(110)+acc71(111)+acc71(112)+acc71(113)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d71h4l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd71h4_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d71
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      Q(1:4)  =cmplx(real(-Q_ext(0:3),  ki_nin), aimag(-Q_ext(0:3)), ki)
      d71 = 0.0_ki
      d71 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d71, ki), aimag(d71), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d71h4l1_qp
