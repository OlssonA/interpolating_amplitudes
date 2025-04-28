module     p2_gg_httbar_d255h12l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d255h12l1_qp.f90
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
      use p2_gg_httbar_abbrevd255h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc255(63)
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspk2
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspval3e1
      complex(ki) :: QspQ
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvae2e1
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspk2 = dotproduct(Q,k2)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspval3e1 = dotproduct(Q,spval3e1)
      QspQ = dotproduct(Q,Q)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      acc255(1)=abb255(7)
      acc255(2)=abb255(8)
      acc255(3)=abb255(9)
      acc255(4)=abb255(10)
      acc255(5)=abb255(11)
      acc255(6)=abb255(12)
      acc255(7)=abb255(13)
      acc255(8)=abb255(14)
      acc255(9)=abb255(15)
      acc255(10)=abb255(16)
      acc255(11)=abb255(17)
      acc255(12)=abb255(18)
      acc255(13)=abb255(19)
      acc255(14)=abb255(20)
      acc255(15)=abb255(21)
      acc255(16)=abb255(22)
      acc255(17)=abb255(23)
      acc255(18)=abb255(24)
      acc255(19)=abb255(25)
      acc255(20)=abb255(26)
      acc255(21)=abb255(27)
      acc255(22)=abb255(28)
      acc255(23)=abb255(29)
      acc255(24)=abb255(30)
      acc255(25)=abb255(31)
      acc255(26)=abb255(32)
      acc255(27)=abb255(33)
      acc255(28)=abb255(34)
      acc255(29)=abb255(35)
      acc255(30)=abb255(36)
      acc255(31)=abb255(37)
      acc255(32)=abb255(38)
      acc255(33)=abb255(39)
      acc255(34)=abb255(40)
      acc255(35)=abb255(45)
      acc255(36)=abb255(46)
      acc255(37)=abb255(47)
      acc255(38)=abb255(49)
      acc255(39)=abb255(50)
      acc255(40)=abb255(51)
      acc255(41)=abb255(52)
      acc255(42)=abb255(55)
      acc255(43)=abb255(56)
      acc255(44)=abb255(58)
      acc255(45)=abb255(59)
      acc255(46)=abb255(60)
      acc255(47)=abb255(62)
      acc255(48)=Qspvae2l4*acc255(1)
      acc255(49)=Qspvae2l5*acc255(22)
      acc255(48)=acc255(49)+acc255(8)+acc255(48)
      acc255(48)=Qspvak2e1*acc255(48)
      acc255(49)=Qspvak1l4*acc255(34)
      acc255(50)=Qspvak1l5*acc255(30)
      acc255(51)=Qspk2*acc255(43)
      acc255(52)=Qspvae2l3*acc255(31)
      acc255(53)=-Qspval3e1*acc255(42)
      acc255(54)=Qspval3e1*acc255(46)
      acc255(54)=acc255(18)+acc255(54)
      acc255(54)=Qspvae2l4*acc255(54)
      acc255(55)=Qspval3e1*acc255(45)
      acc255(55)=acc255(44)+acc255(55)
      acc255(55)=Qspvae2l5*acc255(55)
      acc255(56)=QspQ*acc255(6)
      acc255(48)=acc255(48)+acc255(56)+acc255(55)+acc255(54)+acc255(53)+acc255(&
      &52)+acc255(51)+acc255(50)+acc255(23)+acc255(49)
      acc255(48)=Qspvae1e2*acc255(48)
      acc255(49)=Qspvae1l4*acc255(29)
      acc255(50)=Qspvae1l5*acc255(47)
      acc255(49)=acc255(50)+acc255(33)+acc255(49)
      acc255(49)=QspQ*acc255(49)
      acc255(50)=Qspvak2l4*acc255(25)
      acc255(51)=Qspval3e2*acc255(37)
      acc255(52)=Qspvak2e2*acc255(10)
      acc255(53)=Qspvae1l4*acc255(28)
      acc255(54)=Qspvak2e2*acc255(38)
      acc255(54)=acc255(39)+acc255(54)
      acc255(54)=Qspvae1l5*acc255(54)
      acc255(49)=acc255(49)+acc255(54)+acc255(53)+acc255(52)+acc255(51)+acc255(&
      &24)+acc255(50)
      acc255(49)=Qspvae2e1*acc255(49)
      acc255(50)=Qspvak2l4*acc255(16)
      acc255(51)=Qspval3e2*acc255(7)
      acc255(52)=Qspvak2e2*acc255(15)
      acc255(53)=QspQ*acc255(20)
      acc255(50)=acc255(53)+acc255(52)+acc255(51)+acc255(17)+acc255(50)
      acc255(50)=Qspvak2e1*acc255(50)
      acc255(51)=Qspk2*acc255(41)
      acc255(52)=Qspvae2l3*acc255(9)
      acc255(51)=acc255(52)+acc255(5)+acc255(51)
      acc255(51)=Qspvae1l5*acc255(51)
      acc255(52)=Qspvae1l4*acc255(12)
      acc255(53)=Qspvae1l5*acc255(13)
      acc255(52)=acc255(53)+acc255(14)+acc255(52)
      acc255(52)=QspQ*acc255(52)
      acc255(53)=Qspvak1l4*acc255(32)
      acc255(54)=Qspvak1l5*acc255(26)
      acc255(55)=Qspk2*acc255(19)
      acc255(56)=Qspvak2l4*acc255(21)
      acc255(57)=Qspval3e2*acc255(35)
      acc255(58)=Qspvae2l3*acc255(11)
      acc255(59)=Qspvak2e2*acc255(40)
      acc255(60)=-Qspval3e1*acc255(36)
      acc255(61)=Qspvae1l4*acc255(2)
      acc255(62)=Qspvae2l4*acc255(27)
      acc255(63)=Qspvae2l5*acc255(3)
      brack=acc255(4)+acc255(48)+acc255(49)+acc255(50)+acc255(51)+acc255(52)+ac&
      &c255(53)+acc255(54)+acc255(55)+acc255(56)+acc255(57)+acc255(58)+acc255(5&
      &9)+acc255(60)+acc255(61)+acc255(62)+acc255(63)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d255h12l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd255h12_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d255
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2-k3-k5
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d255 = 0.0_ki
      d255 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d255, ki), aimag(d255), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d255h12l1_qp
