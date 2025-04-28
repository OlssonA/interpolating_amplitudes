module     p2_gg_httbar_d125h8l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d125h8l1.f90
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
      use p2_gg_httbar_abbrevd125h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc125(61)
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae1k2
      complex(ki) :: QspQ
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvae2e1
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      QspQ = dotproduct(Q,Q)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      acc125(1)=abb125(7)
      acc125(2)=abb125(8)
      acc125(3)=abb125(9)
      acc125(4)=abb125(10)
      acc125(5)=abb125(11)
      acc125(6)=abb125(12)
      acc125(7)=abb125(13)
      acc125(8)=abb125(14)
      acc125(9)=abb125(15)
      acc125(10)=abb125(16)
      acc125(11)=abb125(17)
      acc125(12)=abb125(18)
      acc125(13)=abb125(19)
      acc125(14)=abb125(20)
      acc125(15)=abb125(21)
      acc125(16)=abb125(22)
      acc125(17)=abb125(23)
      acc125(18)=abb125(24)
      acc125(19)=abb125(25)
      acc125(20)=abb125(26)
      acc125(21)=abb125(27)
      acc125(22)=abb125(28)
      acc125(23)=abb125(29)
      acc125(24)=abb125(30)
      acc125(25)=abb125(31)
      acc125(26)=abb125(32)
      acc125(27)=abb125(33)
      acc125(28)=abb125(34)
      acc125(29)=abb125(35)
      acc125(30)=abb125(36)
      acc125(31)=abb125(37)
      acc125(32)=abb125(38)
      acc125(33)=abb125(39)
      acc125(34)=abb125(40)
      acc125(35)=abb125(41)
      acc125(36)=abb125(42)
      acc125(37)=abb125(43)
      acc125(38)=abb125(44)
      acc125(39)=abb125(45)
      acc125(40)=abb125(46)
      acc125(41)=abb125(47)
      acc125(42)=abb125(48)
      acc125(43)=abb125(50)
      acc125(44)=abb125(51)
      acc125(45)=abb125(53)
      acc125(46)=abb125(54)
      acc125(47)=abb125(55)
      acc125(48)=abb125(56)
      acc125(49)=Qspvak2e2*acc125(47)
      acc125(50)=Qspvae2k2*acc125(40)
      acc125(51)=Qspval4e1*acc125(1)
      acc125(52)=Qspval4e2*acc125(20)
      acc125(53)=Qspvae1l5*acc125(26)
      acc125(54)=Qspvae2l5*acc125(27)
      acc125(55)=-Qspvak2e1*acc125(44)
      acc125(56)=Qspvae1k2*acc125(18)
      acc125(49)=acc125(56)+acc125(55)+acc125(54)+acc125(53)+acc125(52)+acc125(&
      &51)+acc125(50)+acc125(8)+acc125(49)
      acc125(49)=QspQ*acc125(49)
      acc125(50)=Qspvae1l3*acc125(25)
      acc125(51)=Qspvae2k2*acc125(36)
      acc125(52)=Qspvae1l5*acc125(16)
      acc125(53)=Qspvae2l5*acc125(39)
      acc125(54)=Qspvae2k2*acc125(45)
      acc125(54)=acc125(22)+acc125(54)
      acc125(54)=Qspvae1e2*acc125(54)
      acc125(50)=acc125(54)+acc125(53)+acc125(52)+acc125(51)+acc125(9)+acc125(5&
      &0)
      acc125(50)=Qspvak2e1*acc125(50)
      acc125(51)=Qspval3e1*acc125(32)
      acc125(52)=Qspvak2e2*acc125(7)
      acc125(53)=Qspval4e1*acc125(14)
      acc125(54)=Qspval4e2*acc125(42)
      acc125(55)=Qspvak2e2*acc125(45)
      acc125(55)=acc125(34)+acc125(55)
      acc125(55)=Qspvae2e1*acc125(55)
      acc125(51)=acc125(55)+acc125(54)+acc125(53)+acc125(52)+acc125(11)+acc125(&
      &51)
      acc125(51)=Qspvae1k2*acc125(51)
      acc125(52)=Qspvae2k2*acc125(46)
      acc125(53)=Qspval4e1*acc125(6)
      acc125(54)=-Qspval4e1*acc125(38)
      acc125(54)=acc125(21)+acc125(54)
      acc125(54)=Qspvae2l5*acc125(54)
      acc125(52)=acc125(54)+acc125(53)+acc125(2)+acc125(52)
      acc125(52)=Qspvae1e2*acc125(52)
      acc125(53)=Qspvak2e2*acc125(48)
      acc125(54)=Qspval4e2*acc125(37)
      acc125(55)=-Qspval4e2*acc125(38)
      acc125(55)=acc125(4)+acc125(55)
      acc125(55)=Qspvae1l5*acc125(55)
      acc125(53)=acc125(55)+acc125(54)+acc125(13)+acc125(53)
      acc125(53)=Qspvae2e1*acc125(53)
      acc125(54)=Qspval3e1*acc125(29)
      acc125(55)=Qspval4e2*acc125(35)
      acc125(54)=acc125(55)+acc125(17)+acc125(54)
      acc125(54)=Qspvae1l5*acc125(54)
      acc125(55)=Qspval3e1*acc125(31)
      acc125(56)=Qspval4e1*acc125(10)
      acc125(55)=acc125(56)+acc125(19)+acc125(55)
      acc125(55)=Qspvae2l5*acc125(55)
      acc125(56)=Qspval3e1*acc125(30)
      acc125(57)=Qspvae1l3*acc125(15)
      acc125(58)=Qspvae1l3*acc125(43)
      acc125(58)=acc125(5)+acc125(58)
      acc125(58)=Qspvak2e2*acc125(58)
      acc125(59)=Qspval3e1*acc125(41)
      acc125(59)=acc125(33)+acc125(59)
      acc125(59)=Qspvae2k2*acc125(59)
      acc125(60)=Qspvae1l3*acc125(24)
      acc125(60)=acc125(12)+acc125(60)
      acc125(60)=Qspval4e1*acc125(60)
      acc125(61)=Qspvae1l3*acc125(23)
      acc125(61)=acc125(28)+acc125(61)
      acc125(61)=Qspval4e2*acc125(61)
      brack=acc125(3)+acc125(49)+acc125(50)+acc125(51)+acc125(52)+acc125(53)+ac&
      &c125(54)+acc125(55)+acc125(56)+acc125(57)+acc125(58)+acc125(59)+acc125(6&
      &0)+acc125(61)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d125h8l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd125h8
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d125
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k3-k2+k5
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d125 = 0.0_ki
      d125 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d125, ki), aimag(d125), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d125h8l1
