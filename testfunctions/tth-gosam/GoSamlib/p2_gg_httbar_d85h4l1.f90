module     p2_gg_httbar_d85h4l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d85h4l1.f90
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
      use p2_gg_httbar_abbrevd85h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc85(58)
      complex(ki) :: Qspvae1k2
      complex(ki) :: QspQ
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspe2
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspvae1e2
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      QspQ = dotproduct(Q,Q)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspe2 = dotproduct(Q,e2)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      acc85(1)=abb85(8)
      acc85(2)=abb85(9)
      acc85(3)=abb85(10)
      acc85(4)=abb85(11)
      acc85(5)=abb85(12)
      acc85(6)=abb85(13)
      acc85(7)=abb85(14)
      acc85(8)=abb85(15)
      acc85(9)=abb85(16)
      acc85(10)=abb85(17)
      acc85(11)=abb85(18)
      acc85(12)=abb85(19)
      acc85(13)=abb85(20)
      acc85(14)=abb85(21)
      acc85(15)=abb85(22)
      acc85(16)=abb85(23)
      acc85(17)=abb85(24)
      acc85(18)=abb85(25)
      acc85(19)=abb85(26)
      acc85(20)=abb85(27)
      acc85(21)=abb85(28)
      acc85(22)=abb85(29)
      acc85(23)=abb85(30)
      acc85(24)=abb85(31)
      acc85(25)=abb85(32)
      acc85(26)=abb85(33)
      acc85(27)=abb85(34)
      acc85(28)=abb85(35)
      acc85(29)=abb85(37)
      acc85(30)=abb85(42)
      acc85(31)=abb85(43)
      acc85(32)=abb85(45)
      acc85(33)=abb85(50)
      acc85(34)=abb85(55)
      acc85(35)=abb85(57)
      acc85(36)=abb85(61)
      acc85(37)=abb85(63)
      acc85(38)=abb85(64)
      acc85(39)=abb85(65)
      acc85(40)=abb85(69)
      acc85(41)=abb85(72)
      acc85(42)=abb85(81)
      acc85(43)=abb85(82)
      acc85(44)=acc85(3)*Qspvae1k2
      acc85(45)=acc85(16)*QspQ
      acc85(46)=acc85(17)*Qspval3k1
      acc85(47)=acc85(20)*Qspvak2k1
      acc85(48)=acc85(31)*Qspvae1l5
      acc85(49)=acc85(33)*Qspvae1l4
      acc85(50)=acc85(36)*Qspvae1l3
      acc85(44)=acc85(50)+acc85(49)+acc85(48)+acc85(47)+acc85(46)+acc85(45)+acc&
      &85(44)+acc85(1)
      acc85(44)=Qspvae2e1*acc85(44)
      acc85(45)=acc85(10)*Qspval3e1
      acc85(46)=acc85(13)*Qspvak2e1
      acc85(45)=acc85(46)+acc85(45)+acc85(9)
      acc85(45)=acc85(45)*Qspvae1k2
      acc85(46)=acc85(23)*Qspvae1l4
      acc85(47)=acc85(35)*Qspvae1l3
      acc85(46)=acc85(47)+acc85(30)+acc85(46)
      acc85(46)=acc85(46)*Qspval5e1
      acc85(47)=acc85(8)*Qspvak2e1
      acc85(48)=acc85(29)*Qspval3e1
      acc85(49)=acc85(32)*Qspvae1l4
      acc85(50)=acc85(38)*Qspvae1l3
      acc85(45)=acc85(50)+acc85(49)+acc85(48)+acc85(25)+acc85(47)+acc85(46)+acc&
      &85(45)
      acc85(45)=Qspe2*acc85(45)
      acc85(46)=acc85(11)*Qspvak2e1
      acc85(47)=acc85(18)*Qspval5e1
      acc85(48)=acc85(26)*Qspvak1l4
      acc85(49)=acc85(28)*Qspvak1l3
      acc85(50)=acc85(41)*QspQ
      acc85(51)=acc85(43)*Qspval3e1
      acc85(46)=acc85(51)+acc85(50)+acc85(37)+acc85(49)+acc85(48)+acc85(47)+acc&
      &85(46)
      acc85(46)=Qspvae1e2*acc85(46)
      acc85(47)=acc85(34)*Qspvae1l4
      acc85(48)=acc85(39)*Qspvae1l3
      acc85(47)=acc85(48)+acc85(47)+acc85(2)
      acc85(47)=Qspval5e1*acc85(47)
      acc85(48)=acc85(4)*Qspval3e1
      acc85(49)=acc85(12)*Qspvak2e1
      acc85(48)=acc85(49)+acc85(6)+acc85(48)
      acc85(48)=Qspvae1k2*acc85(48)
      acc85(49)=acc85(5)*Qspvak2e1
      acc85(50)=acc85(14)*Qspval3k1
      acc85(51)=acc85(15)*Qspvae1l3
      acc85(52)=acc85(19)*Qspvak2k1
      acc85(53)=acc85(21)*Qspvak1l4
      acc85(54)=acc85(22)*QspQ
      acc85(55)=acc85(24)*Qspvae1l4
      acc85(56)=acc85(27)*Qspvak1l3
      acc85(57)=acc85(40)*Qspvae1l5
      acc85(58)=acc85(42)*Qspval3e1
      brack=acc85(7)+acc85(44)+acc85(45)+acc85(46)+acc85(47)+acc85(48)+acc85(49&
      &)+acc85(50)+acc85(51)+acc85(52)+acc85(53)+acc85(54)+acc85(55)+acc85(56)+&
      &acc85(57)+acc85(58)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d85h4l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd85h4
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d85
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k5-k2
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d85 = 0.0_ki
      d85 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d85, ki), aimag(d85), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d85h4l1
