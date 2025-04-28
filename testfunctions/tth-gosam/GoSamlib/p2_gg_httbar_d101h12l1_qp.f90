module     p2_gg_httbar_d101h12l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d101h12l1_qp.f90
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
      use p2_gg_httbar_abbrevd101h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc101(54)
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspvak1e1
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvak2e1
      complex(ki) :: QspQ
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvae1k1
      complex(ki) :: Qspvae1l3
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspvak1e1 = dotproduct(Q,spvak1e1)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      QspQ = dotproduct(Q,Q)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvae1k1 = dotproduct(Q,spvae1k1)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      acc101(1)=abb101(7)
      acc101(2)=abb101(8)
      acc101(3)=abb101(9)
      acc101(4)=abb101(10)
      acc101(5)=abb101(11)
      acc101(6)=abb101(12)
      acc101(7)=abb101(13)
      acc101(8)=abb101(14)
      acc101(9)=abb101(15)
      acc101(10)=abb101(16)
      acc101(11)=abb101(17)
      acc101(12)=abb101(18)
      acc101(13)=abb101(19)
      acc101(14)=abb101(20)
      acc101(15)=abb101(21)
      acc101(16)=abb101(22)
      acc101(17)=abb101(23)
      acc101(18)=abb101(24)
      acc101(19)=abb101(25)
      acc101(20)=abb101(26)
      acc101(21)=abb101(27)
      acc101(22)=abb101(29)
      acc101(23)=abb101(30)
      acc101(24)=abb101(31)
      acc101(25)=abb101(32)
      acc101(26)=abb101(33)
      acc101(27)=abb101(34)
      acc101(28)=abb101(35)
      acc101(29)=abb101(36)
      acc101(30)=abb101(37)
      acc101(31)=abb101(38)
      acc101(32)=abb101(39)
      acc101(33)=abb101(40)
      acc101(34)=abb101(41)
      acc101(35)=abb101(42)
      acc101(36)=abb101(43)
      acc101(37)=abb101(44)
      acc101(38)=abb101(45)
      acc101(39)=abb101(46)
      acc101(40)=abb101(47)
      acc101(41)=abb101(48)
      acc101(42)=abb101(49)
      acc101(43)=Qspvak1l4*acc101(31)
      acc101(44)=Qspvak1l5*acc101(42)
      acc101(45)=Qspvak1e1*acc101(28)
      acc101(46)=Qspval3e1*acc101(35)
      acc101(47)=-Qspvak2e1*acc101(27)
      acc101(48)=QspQ*acc101(29)
      acc101(49)=-Qspvak2e1*acc101(39)
      acc101(49)=acc101(17)+acc101(49)
      acc101(49)=Qspvae2l4*acc101(49)
      acc101(50)=-Qspvak2e1*acc101(38)
      acc101(50)=acc101(15)+acc101(50)
      acc101(50)=Qspvae2l5*acc101(50)
      acc101(43)=acc101(50)+acc101(49)+acc101(48)+acc101(47)+acc101(46)+acc101(&
      &45)+acc101(44)+acc101(20)+acc101(43)
      acc101(43)=Qspvae1e2*acc101(43)
      acc101(44)=-acc101(39)*Qspvae1l4
      acc101(45)=-acc101(38)*Qspvae1l5
      acc101(44)=acc101(45)+acc101(36)+acc101(44)
      acc101(44)=Qspvae2e1*acc101(44)
      acc101(45)=Qspvak1l4*acc101(1)
      acc101(46)=Qspvak1l5*acc101(2)
      acc101(47)=Qspvak1e1*acc101(32)
      acc101(48)=Qspval3e1*acc101(26)
      acc101(49)=Qspvak2e1*acc101(10)
      acc101(50)=QspQ*acc101(18)
      acc101(44)=acc101(44)+acc101(50)+acc101(49)+acc101(48)+acc101(47)+acc101(&
      &46)+acc101(3)+acc101(45)
      acc101(44)=Qspvak2e2*acc101(44)
      acc101(45)=Qspvak2k1*acc101(37)
      acc101(46)=Qspvae1k1*acc101(40)
      acc101(47)=Qspvae1l3*acc101(24)
      acc101(48)=QspQ*acc101(12)
      acc101(45)=acc101(48)+acc101(47)+acc101(46)+acc101(8)+acc101(45)
      acc101(45)=Qspvae2l4*acc101(45)
      acc101(46)=Qspvak2k1*acc101(30)
      acc101(47)=Qspvae1k1*acc101(41)
      acc101(48)=Qspvae1l3*acc101(34)
      acc101(49)=QspQ*acc101(33)
      acc101(46)=acc101(49)+acc101(48)+acc101(47)+acc101(4)+acc101(46)
      acc101(46)=Qspvae2l5*acc101(46)
      acc101(47)=Qspvak2k1*acc101(22)
      acc101(48)=Qspvae1k1*acc101(21)
      acc101(49)=Qspvae1l3*acc101(16)
      acc101(50)=QspQ*acc101(23)
      acc101(47)=acc101(50)+acc101(49)+acc101(48)+acc101(13)+acc101(47)
      acc101(47)=Qspvae2e1*acc101(47)
      acc101(48)=Qspvae1l4*acc101(14)
      acc101(49)=Qspvae1l5*acc101(11)
      acc101(50)=Qspvak2k1*acc101(9)
      acc101(51)=Qspvae1k1*acc101(5)
      acc101(52)=Qspvae1l3*acc101(25)
      acc101(53)=Qspvak2e1*acc101(19)
      acc101(54)=QspQ*acc101(7)
      brack=acc101(6)+acc101(43)+acc101(44)+acc101(45)+acc101(46)+acc101(47)+ac&
      &c101(48)+acc101(49)+acc101(50)+acc101(51)+acc101(52)+acc101(53)+acc101(5&
      &4)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d101h12l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd101h12_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d101
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2-k4
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d101 = 0.0_ki
      d101 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d101, ki), aimag(d101), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d101h12l1_qp
