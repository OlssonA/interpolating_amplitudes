module     p2_gg_httbar_d113h8l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d113h8l1_qp.f90
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
      use p2_gg_httbar_abbrevd113h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc113(65)
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspk2
      complex(ki) :: Qspvak2e1
      complex(ki) :: QspQ
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspvae2e1
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspk2 = dotproduct(Q,k2)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      QspQ = dotproduct(Q,Q)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      acc113(1)=abb113(7)
      acc113(2)=abb113(8)
      acc113(3)=abb113(9)
      acc113(4)=abb113(10)
      acc113(5)=abb113(11)
      acc113(6)=abb113(12)
      acc113(7)=abb113(13)
      acc113(8)=abb113(15)
      acc113(9)=abb113(16)
      acc113(10)=abb113(17)
      acc113(11)=abb113(18)
      acc113(12)=abb113(19)
      acc113(13)=abb113(20)
      acc113(14)=abb113(21)
      acc113(15)=abb113(22)
      acc113(16)=abb113(23)
      acc113(17)=abb113(24)
      acc113(18)=abb113(25)
      acc113(19)=abb113(26)
      acc113(20)=abb113(27)
      acc113(21)=abb113(28)
      acc113(22)=abb113(29)
      acc113(23)=abb113(30)
      acc113(24)=abb113(31)
      acc113(25)=abb113(32)
      acc113(26)=abb113(33)
      acc113(27)=abb113(34)
      acc113(28)=abb113(35)
      acc113(29)=abb113(36)
      acc113(30)=abb113(37)
      acc113(31)=abb113(38)
      acc113(32)=abb113(39)
      acc113(33)=abb113(40)
      acc113(34)=abb113(41)
      acc113(35)=abb113(42)
      acc113(36)=abb113(43)
      acc113(37)=abb113(44)
      acc113(38)=abb113(45)
      acc113(39)=abb113(46)
      acc113(40)=abb113(48)
      acc113(41)=abb113(50)
      acc113(42)=abb113(51)
      acc113(43)=abb113(52)
      acc113(44)=abb113(53)
      acc113(45)=abb113(54)
      acc113(46)=abb113(55)
      acc113(47)=abb113(56)
      acc113(48)=abb113(57)
      acc113(49)=abb113(58)
      acc113(50)=Qspval4k2*acc113(12)
      acc113(51)=Qspvae2l5*acc113(32)
      acc113(52)=Qspvae2k2*acc113(1)
      acc113(53)=Qspvae2l3*acc113(46)
      acc113(54)=Qspk2*acc113(8)
      acc113(55)=Qspvae2k2*acc113(35)
      acc113(55)=acc113(34)+acc113(55)
      acc113(55)=Qspvak2e1*acc113(55)
      acc113(56)=QspQ*acc113(10)
      acc113(57)=-Qspvae2l5*acc113(41)
      acc113(57)=acc113(42)+acc113(57)
      acc113(57)=Qspval4e1*acc113(57)
      acc113(50)=acc113(57)+acc113(56)+acc113(55)+acc113(54)+acc113(53)+acc113(&
      &52)+acc113(51)+acc113(7)+acc113(50)
      acc113(50)=Qspvae1e2*acc113(50)
      acc113(51)=Qspvak2l5*acc113(38)
      acc113(52)=Qspval4e2*acc113(44)
      acc113(53)=Qspvak2e2*acc113(4)
      acc113(54)=Qspval3e2*acc113(48)
      acc113(55)=Qspk2*acc113(37)
      acc113(56)=Qspvak2e2*acc113(35)
      acc113(56)=acc113(24)+acc113(56)
      acc113(56)=Qspvae1k2*acc113(56)
      acc113(57)=QspQ*acc113(28)
      acc113(58)=-Qspval4e2*acc113(41)
      acc113(58)=acc113(43)+acc113(58)
      acc113(58)=Qspvae1l5*acc113(58)
      acc113(51)=acc113(58)+acc113(57)+acc113(56)+acc113(55)+acc113(54)+acc113(&
      &53)+acc113(52)+acc113(20)+acc113(51)
      acc113(51)=Qspvae2e1*acc113(51)
      acc113(52)=Qspvak2l5*acc113(2)
      acc113(53)=Qspvak2e2*acc113(13)
      acc113(54)=Qspval3e2*acc113(49)
      acc113(55)=QspQ*acc113(45)
      acc113(52)=acc113(55)+acc113(54)+acc113(53)+acc113(31)+acc113(52)
      acc113(52)=Qspval4e1*acc113(52)
      acc113(53)=Qspval4k2*acc113(5)
      acc113(54)=Qspvae2k2*acc113(18)
      acc113(55)=Qspvae2l3*acc113(47)
      acc113(56)=QspQ*acc113(40)
      acc113(53)=acc113(56)+acc113(55)+acc113(54)+acc113(36)+acc113(53)
      acc113(53)=Qspvae1l5*acc113(53)
      acc113(54)=Qspval3e2*acc113(39)
      acc113(55)=Qspk2*acc113(27)
      acc113(54)=acc113(55)+acc113(17)+acc113(54)
      acc113(54)=Qspvak2e1*acc113(54)
      acc113(55)=Qspvae2l3*acc113(11)
      acc113(56)=Qspk2*acc113(15)
      acc113(55)=acc113(56)+acc113(14)+acc113(55)
      acc113(55)=Qspvae1k2*acc113(55)
      acc113(56)=-Qspvak2e1*acc113(9)
      acc113(57)=-Qspvae1k2*acc113(19)
      acc113(56)=acc113(57)+acc113(16)+acc113(56)
      acc113(56)=QspQ*acc113(56)
      acc113(57)=Qspvak2l5*acc113(25)
      acc113(58)=Qspval4k2*acc113(29)
      acc113(59)=Qspval4e2*acc113(33)
      acc113(60)=Qspvae2l5*acc113(21)
      acc113(61)=Qspvak2e2*acc113(6)
      acc113(62)=Qspvae2k2*acc113(23)
      acc113(63)=Qspval3e2*acc113(26)
      acc113(64)=Qspvae2l3*acc113(30)
      acc113(65)=Qspk2*acc113(22)
      brack=acc113(3)+acc113(50)+acc113(51)+acc113(52)+acc113(53)+acc113(54)+ac&
      &c113(55)+acc113(56)+acc113(57)+acc113(58)+acc113(59)+acc113(60)+acc113(6&
      &1)+acc113(62)+acc113(63)+acc113(64)+acc113(65)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d113h8l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd113h8_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d113
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k2+k3+k5
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d113 = 0.0_ki
      d113 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d113, ki), aimag(d113), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d113h8l1_qp
