module     p2_gg_httbar_d90h8l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d90h8l1.f90
   ! generator: buildfortran.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd90h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc90(90)
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspk2
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspe2
      complex(ki) :: Qspl4
      complex(ki) :: QspQ
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspe1
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae1k1
      complex(ki) :: Qspvak1e1
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspk1
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspvak1k2
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspk2 = dotproduct(Q,k2)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspe2 = dotproduct(Q,e2)
      Qspl4 = dotproduct(Q,l4)
      QspQ = dotproduct(Q,Q)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspe1 = dotproduct(Q,e1)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae1k1 = dotproduct(Q,spvae1k1)
      Qspvak1e1 = dotproduct(Q,spvak1e1)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspk1 = dotproduct(Q,k1)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      acc90(1)=abb90(8)
      acc90(2)=abb90(9)
      acc90(3)=abb90(10)
      acc90(4)=abb90(11)
      acc90(5)=abb90(12)
      acc90(6)=abb90(13)
      acc90(7)=abb90(14)
      acc90(8)=abb90(15)
      acc90(9)=abb90(16)
      acc90(10)=abb90(17)
      acc90(11)=abb90(18)
      acc90(12)=abb90(19)
      acc90(13)=abb90(20)
      acc90(14)=abb90(21)
      acc90(15)=abb90(22)
      acc90(16)=abb90(23)
      acc90(17)=abb90(24)
      acc90(18)=abb90(25)
      acc90(19)=abb90(26)
      acc90(20)=abb90(27)
      acc90(21)=abb90(28)
      acc90(22)=abb90(29)
      acc90(23)=abb90(30)
      acc90(24)=abb90(31)
      acc90(25)=abb90(32)
      acc90(26)=abb90(33)
      acc90(27)=abb90(34)
      acc90(28)=abb90(35)
      acc90(29)=abb90(36)
      acc90(30)=abb90(37)
      acc90(31)=abb90(38)
      acc90(32)=abb90(39)
      acc90(33)=abb90(40)
      acc90(34)=abb90(41)
      acc90(35)=abb90(42)
      acc90(36)=abb90(43)
      acc90(37)=abb90(44)
      acc90(38)=abb90(46)
      acc90(39)=abb90(47)
      acc90(40)=abb90(48)
      acc90(41)=abb90(50)
      acc90(42)=abb90(51)
      acc90(43)=abb90(53)
      acc90(44)=abb90(54)
      acc90(45)=abb90(55)
      acc90(46)=abb90(56)
      acc90(47)=abb90(57)
      acc90(48)=abb90(58)
      acc90(49)=abb90(59)
      acc90(50)=abb90(60)
      acc90(51)=abb90(61)
      acc90(52)=abb90(62)
      acc90(53)=abb90(63)
      acc90(54)=abb90(64)
      acc90(55)=abb90(65)
      acc90(56)=abb90(66)
      acc90(57)=abb90(67)
      acc90(58)=abb90(68)
      acc90(59)=abb90(69)
      acc90(60)=abb90(70)
      acc90(61)=abb90(71)
      acc90(62)=abb90(73)
      acc90(63)=abb90(77)
      acc90(64)=abb90(78)
      acc90(65)=abb90(80)
      acc90(66)=abb90(81)
      acc90(67)=abb90(82)
      acc90(68)=abb90(83)
      acc90(69)=abb90(86)
      acc90(70)=abb90(87)
      acc90(71)=abb90(89)
      acc90(72)=acc90(15)*Qspval4l5
      acc90(73)=acc90(25)*Qspval3k2
      acc90(74)=acc90(38)*Qspk2
      acc90(75)=acc90(62)*Qspval4l3
      acc90(72)=acc90(75)+acc90(74)+acc90(73)+acc90(72)+acc90(4)
      acc90(72)=acc90(72)*Qspe2
      acc90(73)=acc90(12)*Qspl4
      acc90(74)=acc90(14)*QspQ
      acc90(75)=acc90(21)*Qspk2
      acc90(76)=acc90(24)*Qspval4k2
      acc90(77)=acc90(28)*Qspval3l4
      acc90(78)=acc90(51)*Qspval4l5
      acc90(79)=acc90(55)*Qspvak2l3
      acc90(80)=acc90(61)*Qspval4l3
      acc90(81)=-acc90(63)*Qspvak2l5
      acc90(82)=acc90(69)*Qspvak2l4
      acc90(83)=Qspvae2l5*acc90(56)
      acc90(84)=Qspval4e2*acc90(64)
      acc90(85)=Qspvae2l3*acc90(53)
      acc90(86)=Qspval3e2*acc90(65)
      acc90(87)=Qspvae2k2*acc90(68)
      acc90(88)=Qspvak2e2*acc90(57)
      acc90(89)=Qspvae2k1*acc90(33)
      acc90(90)=Qspvak1e2*acc90(35)
      acc90(72)=acc90(80)+acc90(78)+acc90(75)+acc90(72)+acc90(90)+acc90(89)+acc&
      &90(88)+acc90(87)+acc90(86)+acc90(85)+acc90(84)+acc90(83)+acc90(82)+acc90&
      &(81)+acc90(79)+acc90(77)+acc90(76)+acc90(74)+acc90(73)+acc90(1)
      acc90(72)=Qspe1*acc90(72)
      acc90(73)=acc90(7)*QspQ
      acc90(74)=acc90(16)*Qspval4k2
      acc90(75)=acc90(17)*Qspk2
      acc90(76)=-acc90(32)*Qspl4
      acc90(77)=acc90(39)*Qspvak2l5
      acc90(78)=acc90(44)*Qspval3l4
      acc90(79)=acc90(47)*Qspvak2l4
      acc90(80)=acc90(49)*Qspvak2l3
      acc90(81)=-acc90(70)*Qspval4l3
      acc90(82)=-acc90(71)*Qspval4l5
      acc90(83)=Qspvae1l5*acc90(59)
      acc90(84)=Qspval4e1*acc90(58)
      acc90(85)=Qspvae1l3*acc90(66)
      acc90(86)=Qspval3e1*acc90(67)
      acc90(87)=Qspvae1k2*acc90(11)
      acc90(88)=Qspvak2e1*acc90(60)
      acc90(89)=Qspvae1k1*acc90(30)
      acc90(90)=Qspvak1e1*acc90(42)
      acc90(73)=acc90(82)+acc90(81)+acc90(75)+acc90(90)+acc90(89)+acc90(88)+acc&
      &90(87)+acc90(86)+acc90(85)+acc90(84)+acc90(83)+acc90(80)+acc90(79)+acc90&
      &(46)+acc90(78)+acc90(77)+acc90(76)+acc90(74)+acc90(73)
      acc90(73)=Qspe2*acc90(73)
      acc90(74)=-acc90(20)*Qspk2
      acc90(75)=acc90(37)*Qspvae2e1
      acc90(76)=acc90(43)*Qspval4l3
      acc90(77)=acc90(50)*Qspvae1e2
      acc90(78)=acc90(54)*Qspval4l5
      acc90(74)=acc90(78)+acc90(77)+acc90(76)+acc90(75)+acc90(74)+acc90(2)
      acc90(74)=QspQ*acc90(74)
      acc90(75)=acc90(6)*Qspvak2l5
      acc90(76)=acc90(8)*Qspvak2l3
      acc90(77)=acc90(19)*Qspvak2l4
      acc90(75)=acc90(77)+acc90(13)+acc90(76)+acc90(75)
      acc90(75)=Qspval4k2*acc90(75)
      acc90(76)=-acc90(23)*Qspvae2e1
      acc90(77)=acc90(36)*Qspvae1e2
      acc90(76)=acc90(77)+acc90(76)
      acc90(77)=Qspk1-Qspk2
      acc90(76)=acc90(77)*acc90(76)
      acc90(77)=acc90(34)*Qspval4l3
      acc90(78)=acc90(52)*Qspval4l5
      acc90(77)=acc90(78)+acc90(77)+acc90(10)
      acc90(77)=Qspl4*acc90(77)
      acc90(78)=-Qspval3k2*QspQ
      acc90(79)=-Qspval3l4*Qspval4k2
      acc90(78)=acc90(78)+acc90(79)
      acc90(78)=acc90(26)*acc90(78)
      acc90(79)=-acc90(22)*Qspk2
      acc90(79)=acc90(79)+acc90(5)
      acc90(79)=Qspval3k2*acc90(79)
      acc90(80)=-acc90(3)*Qspk2**2
      acc90(81)=acc90(9)*Qspk2
      acc90(82)=acc90(27)*Qspvae2e1
      acc90(83)=acc90(29)*Qspk1
      acc90(84)=acc90(41)*Qspvae1e2
      acc90(85)=acc90(45)*Qspval4l3
      acc90(86)=acc90(48)*Qspval4l5
      acc90(87)=Qspval4k1*acc90(31)
      acc90(88)=Qspvak1k2*acc90(18)
      brack=acc90(40)+acc90(72)+acc90(73)+acc90(74)+acc90(75)+acc90(76)+acc90(7&
      &7)+acc90(78)+acc90(79)+acc90(80)+acc90(81)+acc90(82)+acc90(83)+acc90(84)&
      &+acc90(85)+acc90(86)+acc90(87)+acc90(88)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d90h8l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd90h8
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d90
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k2
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d90 = 0.0_ki
      d90 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d90, ki), aimag(d90), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d90h8l1
